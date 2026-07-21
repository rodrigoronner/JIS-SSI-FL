// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title AccessControl
 * @dev DEPRECATED / UNUSED: superseded by FLRegistry.sol, which is the
 * contract actually deployed by scripts/deploy.js and used throughout the
 * simulation. This file is an earlier draft kept only for reference and
 * is safe to delete.
 */
contract AccessControl {
    address public issuer;
    mapping(address => bool) public registry;
    
    event UpdateAccepted(uint round, string modelHash, address indexed worker);
    event WorkerAuthorized(address indexed worker);

    constructor() {
        issuer = msg.sender;
    }

    modifier onlyIssuer() {
        require(msg.sender == issuer, "Only issuer can perform this action");
        _;
    }

    function authorizeWorker(address worker) public onlyIssuer {
        registry[worker] = true;
        emit WorkerAuthorized(worker);
    }

    function submitUpdate(string memory modelHash) public {
        require(registry[msg.sender], "Access Denied: No Valid VC");
        // Log the hash for off-chain aggregation verification
        emit UpdateAccepted(block.timestamp, modelHash, msg.sender);
    }

    function isAuthorized(address worker) public view returns (bool) {
        return registry[worker];
    }
}
