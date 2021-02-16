// SPDX-License-Identifier: agpl-3.0
pragma solidity 0.6.12;

import {ILendingPool} from '../../interfaces/ILendingPool.sol';
import {IERC20} from '../../dependencies/openzeppelin/contracts/IERC20.sol';
import {IAToken} from '../../interfaces/IAToken.sol';
import {ERC20} from '../../dependencies/openzeppelin/contracts/ERC20.sol';
import {SafeERC20} from '../../dependencies/openzeppelin/contracts/SafeERC20.sol';
import {WadRayMath} from '../../protocol/libraries/math/WadRayMath.sol';

/**
 * @title StaticAToken
 * @dev Wrapper token that allows to deposit tokens on the Aave protocol and receive
 * a token which balance doesn't increase automatically, but uses an ever-increasing exchange rate
 * - Only supporting deposits and withdrawals
 * @author Aave
 **/
contract StaticAToken is ERC20 {
  using SafeERC20 for IERC20;
  using WadRayMath for uint256;

  bytes public constant EIP712_REVISION = bytes('1');
  bytes32 internal constant EIP712_DOMAIN =
    keccak256('EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)');
  bytes32 public constant PERMIT_TYPEHASH =
    keccak256('Permit(address owner,address spender,uint256 value,uint256 nonce,uint256 deadline)');
  bytes32 public constant METADEPOSIT_TYPEHASH =
    keccak256(
      'Deposit(address depositor,address recipient,uint256 value,uint16 referralCode,bool fromUnderlying,uint256 nonce,uint256 deadline)'
    );
  bytes32 public constant METAWITHDRAWAL_TYPEHASH =
    keccak256(
      'Withdraw(address owner,address recipient,uint256 value, bool toUnderlying, uint256 nonce,uint256 deadline)'
    );

  ILendingPool public immutable LENDING_POOL;
  IERC20 public immutable ATOKEN;
  IERC20 public immutable ASSET;

  /// @dev owner => next valid nonce to submit with permit(), metaDeposit() and metaWithdraw()
  /// We choose to have sequentiality on them for each user to avoid potentially dangerous/bad UX cases
  mapping(address => uint256) public _nonces;

  constructor(
    ILendingPool lendingPool,
    address aToken,
    string memory wrappedTokenName,
    string memory wrappedTokenSymbol
  ) public ERC20(wrappedTokenName, wrappedTokenSymbol) {
    LENDING_POOL = lendingPool;
    ATOKEN = IERC20(aToken);

    IERC20 underlyingAsset = IERC20(IAToken(aToken).UNDERLYING_ASSET_ADDRESS());
    ASSET = underlyingAsset;
    underlyingAsset.approve(address(lendingPool), type(uint256).max);
  }

  /**
   * @dev Deposits `ASSET` in the Aave protocol and mints static aTokens to msg.sender
   * @param recipient The address that will receive the static aTokens
   * @param amount The amount of underlying `ASSET` to deposit (e.g. deposit of 100 USDC)
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   * @param fromUnderlying bool
   * - `true` if the msg.sender comes with underlying tokens (e.g. USDC)
   * - `false` if the msg.sender comes already with aTokens (e.g. aUSDC)
   **/
  function deposit(
    address recipient,
    uint256 amount,
    uint16 referralCode,
    bool fromUnderlying
  ) external {
    _deposit(msg.sender, recipient, amount, referralCode, fromUnderlying);
  }

  /**
   * @dev Burns `amount` of static aToken, with recipient receiving the corresponding amount of `ASSET`
   * @param recipient The address that will receive the amount of `ASSET` withdrawn from the Aave protocol
   * @param amount The amount of static aToken to burn
   * @param toUnderlying bool
   * - `true` for the recipient to get underlying tokens (e.g. USDC)
   * - `false` for the recipient to get aTokens (e.g. aUSDC)
   **/
  function withdraw(
    address recipient,
    uint256 amount,
    bool toUnderlying
  ) external {
    _withdraw(msg.sender, recipient, amount, toUnderlying);
  }

  /**
   * @dev Implements the permit function as for
   * https://github.com/ethereum/EIPs/blob/8a34d644aacf0f9f8f00815307fd7dd5da07655f/EIPS/eip-2612.md
   * @param owner The owner of the funds
   * @param spender The spender
   * @param value The amount
   * @param deadline The deadline timestamp, type(uint256).max for max deadline
   * @param v Signature param
   * @param s Signature param
   * @param r Signature param
   * @param chainId Passing the chainId in order to be fork-compatible
   */
  function permit(
    address owner,
    address spender,
    uint256 value,
    uint256 deadline,
    uint8 v,
    bytes32 r,
    bytes32 s,
    uint256 chainId
  ) external {
    require(owner != address(0), 'INVALID_OWNER');
    //solium-disable-next-line
    require(block.timestamp <= deadline, 'INVALID_EXPIRATION');
    uint256 currentValidNonce = _nonces[owner];
    bytes32 digest =
      keccak256(
        abi.encodePacked(
          '\x19\x01',
          getDomainSeparator(chainId),
          keccak256(abi.encode(PERMIT_TYPEHASH, owner, spender, value, currentValidNonce, deadline))
        )
      );
    require(owner == ecrecover(digest, v, r, s), 'INVALID_SIGNATURE');
    _nonces[owner] = currentValidNonce.add(1);
    _approve(owner, spender, value);
  }

  /**
   * @dev Allows to deposit on Aave via meta-transaction
   * https://github.com/ethereum/EIPs/blob/8a34d644aacf0f9f8f00815307fd7dd5da07655f/EIPS/eip-2612.md
   * @param depositor Address from which the funds to deposit are going to be pulled
   * @param recipient Address that will receive the staticATokens, in the average case, same as the `depositor`
   * @param value The amount to deposit
   * @param referralCode Code used to register the integrator originating the operation, for potential rewards.
   *   0 if the action is executed directly by the user, without any middle-man
   * @param fromUnderlying bool
   * - `true` if the msg.sender comes with underlying tokens (e.g. USDC)
   * - `false` if the msg.sender comes already with aTokens (e.g. aUSDC)
   * @param deadline The deadline timestamp, type(uint256).max for max deadline
   * @param v Signature param
   * @param s Signature param
   * @param r Signature param
   * @param chainId Passing the chainId in order to be fork-compatible
   */
  function metaDeposit(
    address depositor,
    address recipient,
    uint256 value,
    uint16 referralCode,
    bool fromUnderlying,
    uint256 deadline,
    uint8 v,
    bytes32 r,
    bytes32 s,
    uint256 chainId
  ) external {
    require(depositor != address(0), 'INVALID_DEPOSITOR');
    //solium-disable-next-line
    require(block.timestamp <= deadline, 'INVALID_EXPIRATION');
    uint256 currentValidNonce = _nonces[depositor];
    bytes32 digest =
      keccak256(
        abi.encodePacked(
          '\x19\x01',
          getDomainSeparator(chainId),
          keccak256(
            abi.encode(
              METADEPOSIT_TYPEHASH,
              depositor,
              recipient,
              value,
              referralCode,
              fromUnderlying,
              currentValidNonce,
              deadline
            )
          )
        )
      );
    require(depositor == ecrecover(digest, v, r, s), 'INVALID_SIGNATURE');
    _nonces[depositor] = currentValidNonce.add(1);
    _deposit(depositor, recipient, value, referralCode, fromUnderlying);
  }

  /**
   * @dev Allows to withdraw from Aave via meta-transaction
   * https://github.com/ethereum/EIPs/blob/8a34d644aacf0f9f8f00815307fd7dd5da07655f/EIPS/eip-2612.md
   * @param owner Address owning the staticATokens
   * @param recipient Address that will receive the underlying withdrawn from Aave
   * @param value The amount of staticAToken to withdraw
   * @param toUnderlying bool
   * - `true` for the recipient to get underlying tokens (e.g. USDC)
   * - `false` for the recipient to get aTokens (e.g. aUSDC)
   * @param deadline The deadline timestamp, type(uint256).max for max deadline
   * @param v Signature param
   * @param s Signature param
   * @param r Signature param
   * @param chainId Passing the chainId in order to be fork-compatible
   */
  function metaWithdraw(
    address owner,
    address recipient,
    uint256 value,
    bool toUnderlying,
    uint256 deadline,
    uint8 v,
    bytes32 r,
    bytes32 s,
    uint256 chainId
  ) external {
    require(owner != address(0), 'INVALID_DEPOSITOR');
    //solium-disable-next-line
    require(block.timestamp <= deadline, 'INVALID_EXPIRATION');
    uint256 currentValidNonce = _nonces[owner];
    bytes32 digest =
      keccak256(
        abi.encodePacked(
          '\x19\x01',
          getDomainSeparator(chainId),
          keccak256(
            abi.encode(
              METAWITHDRAWAL_TYPEHASH,
              owner,
              recipient,
              value,
              toUnderlying,
              currentValidNonce,
              deadline
            )
          )
        )
      );
    require(owner == ecrecover(digest, v, r, s), 'INVALID_SIGNATURE');
    _nonces[owner] = currentValidNonce.add(1);
    _withdraw(owner, recipient, value, toUnderlying);
  }

  /**
   * @dev Utility method to get the current aToken balance of an user, from his staticAToken balance
   * @param account The address of the user
   * @return uint256 The aToken balance
   **/
  function dynamicBalanceOf(address account) external view returns (uint256) {
    return staticToDynamicAmount(balanceOf(account));
  }

  /**
   * @dev Converts a static amount (scaled balance on aToken) to the aToken/underlying value,
   * using the current liquidity index on Aave
   * @param amount The amount to convert from
   * @return uint256 The dynamic amount
   **/
  function staticToDynamicAmount(uint256 amount) public view returns (uint256) {
    return amount.rayMul(rate());
  }

  /**
   * @dev Converts an aToken or underlying amount to the what it is denominated on the aToken as
   * scaled balance, function of the principal and the liquidity index
   * @param amount The amount to convert from
   * @return uint256 The static (scaled) amount
   **/
  function dynamicToStaticAmount(uint256 amount) public view returns (uint256) {
    return amount.rayDiv(rate());
  }

  /**
   * @dev Returns the Aave liquidity index of the underlying aToken, denominated rate here
   * as it can be considered as an ever-increasing exchange rate
   * @return bytes32 The domain separator
   **/
  function rate() public view returns (uint256) {
    return LENDING_POOL.getReserveNormalizedIncome(address(ASSET));
  }

  /**
   * @dev Function to return a dynamic domain separator, in order to be compatible with forks changing chainId
   * @param chainId The chain id
   * @return bytes32 The domain separator
   **/
  function getDomainSeparator(uint256 chainId) public view returns (bytes32) {
    return
      keccak256(
        abi.encode(
          EIP712_DOMAIN,
          keccak256(bytes(name())),
          keccak256(EIP712_REVISION),
          chainId,
          address(this)
        )
      );
  }

  function _deposit(
    address depositor,
    address recipient,
    uint256 amount,
    uint16 referralCode,
    bool fromUnderlying
  ) internal {
    require(recipient != address(0), 'INVALID_RECIPIENT');

    if (fromUnderlying) {
      ASSET.safeTransferFrom(depositor, address(this), amount);
      LENDING_POOL.deposit(address(ASSET), amount, address(this), referralCode);
    } else {
      ATOKEN.safeTransferFrom(depositor, address(this), amount);
    }

    _mint(recipient, dynamicToStaticAmount(amount));
  }

  function _withdraw(
    address owner,
    address recipient,
    uint256 amount,
    bool toUnderlying
  ) internal {
    require(recipient != address(0), 'INVALID_RECIPIENT');

    uint256 userBalance = balanceOf(owner);
    uint256 amountToWithdraw = (amount > userBalance) ? userBalance : amount;

    _burn(owner, amountToWithdraw);

    if (toUnderlying) {
      LENDING_POOL.withdraw(address(ASSET), staticToDynamicAmount(amountToWithdraw), recipient);
    } else {
      ATOKEN.safeTransfer(recipient, staticToDynamicAmount(amountToWithdraw));
    }
  }
}
