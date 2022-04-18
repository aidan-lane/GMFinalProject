rm -rf ./gen/joyride_pb2_grpc.py
rm -rf ./gen/joyride_pb2.py
python3 -m grpc_tools.protoc -I./protos --python_out=./gen --grpc_python_out=./gen ./protos/joyride.proto
echo Generated GRPC Files