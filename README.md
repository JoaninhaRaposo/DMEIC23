# Overcooked Demo


Adapation of Overcooked Demo where humans can play with trained AI agents. 
* [Installation](#installation)
* [Usage](#usage)
* [Dependencies](#dependencies)
* [Configuration](#configuration)

## Installation

Building the server image requires [Docker](https://docs.docker.com/get-docker/)

## Usage

In order to build and run the development server, run
```bash
./up.sh
```

After running one of the above commands, navigate to http://localhost (PARAMS changed - adjust them)

In order to kill the production server, run
```bash
./down.sh
```

## Dependencies

The Overcooked-Demo server relies on some repositories. Both repos are automatically cloned and installed in the Docker builds.



## Configuration

Basic game settings can be configured by changing the values in [config.json](server/config.json)
