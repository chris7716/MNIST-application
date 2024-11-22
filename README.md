# MNIST Application

This README will guide you through setting up a MNIST application using Docker and Docker Compose. The setup will allow you to quickly get the Flask app running with just a few commands.

---

## Architecture

![Architecture!](/static/architecture.png "Architecture")

## Prerequisites

Before you begin, ensure you have the following tools installed on your machine:

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)
- [Git](https://git-scm.com/)

Update the .env file with DB configs/

You can verify that Docker and Docker Compose are installed by running:

```bash
docker --version
docker-compose --version
```

Finally run the Docker compose

```bash
docker-compose up -d --build
```
