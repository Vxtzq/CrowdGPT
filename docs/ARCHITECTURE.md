# System architecture
The CrowdGPT framework is Split into two parts:
- The server
- The client
**The server code is currently closed source for sécurité reasons**
The server however exposes a simple api that allows any compatible client (having a GUI or not) to contribute

## Server architecture
The server acts as a centralized coordinator that assembles all the clients contributions, and merge these on the main model.

## Client architecture
The client is a simple pytorch training loop wrapper that trains the main model on provided data. It then bring back its updates on the main model to the server, which savez these updates after fureter verification.
