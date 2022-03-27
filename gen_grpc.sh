rm -rf ./gen/*
python -m grpc_tools.protoc -I./protos --python_out=./gen --grpc_python_out=./gen ./protos/joyride.proto
echo Generated GRPC Files