rm -rf ./gen/*
python -m grpc_tools.protoc -I./protos --python_out=./gen --grpc_python_out=./gen ./protos/route_guide.proto
echo Generated GRPC Files