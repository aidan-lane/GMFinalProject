# Joy Ride

## Setup
Run `pip3 install -r requirements.txt` to install all needed Python3 dependencies.

Make sure to run `./grpc.sh` whenever a proto is modified. This re-generates needed GRPC files
for the Python client and server.

### Application:
Run `./app.sh` to start the application.

### Database:
Run `docker compose up` to start the Postgres database.

### Server:
Run `./server.sh`.