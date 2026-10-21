
<img src="https://raw.githubusercontent.com/Vxtzq/CrowdGPT/refs/heads/main/docs/diagram.jpeg" alt="The CrowdGPT architecture"/>


# System architecture
The CrowdGPT framework is Split into two parts:

- The server
- The client

The server exposes a simple api that allows any compatible client (having a GUI or not) to contribute

## Server architecture
The server acts as a centralized coordinator that assembles all the clients contributions, and merge these on the main model.

## Client architecture
The client is a simple pytorch training loop wrapper that trains the main model on provided data. It then bring back its updates on the main model to the server, which saves these updates after further verification.

The client trains for 1 hour, then sends back the trained model to the server as a weight delta.

**The client is therefore 100% auditable, opensource, and safe to use** (no arbitrary code coming from server is ever executed)
