# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: joyride.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\rjoyride.proto\x12\x07joyride\"7\n\x0bRideRequest\x12\r\n\x05start\x18\x01 \x01(\t\x12\x0b\n\x03\x65nd\x18\x02 \x01(\t\x12\x0c\n\x04time\x18\x03 \x01(\x05\"*\n\tRideReply\x12\x0c\n\x04node\x18\x01 \x01(\x03\x12\x0f\n\x07message\x18\x02 \x01(\t\"*\n\nRideRating\x12\x0e\n\x06rating\x18\x01 \x01(\x03\x12\x0c\n\x04\x64\x61ta\x18\x02 \x03(\x03\"\x06\n\x04Null2|\n\x07JoyRide\x12:\n\nGetJoyRide\x12\x14.joyride.RideRequest\x1a\x12.joyride.RideReply\"\x00\x30\x01\x12\x35\n\rGetRideRating\x12\x13.joyride.RideRating\x1a\r.joyride.Null\"\x00\x42\x30\n\x18io.grpc.examples.joyrideB\x0cJoyrideProtoP\x01\xa2\x02\x03HLWb\x06proto3')



_RIDEREQUEST = DESCRIPTOR.message_types_by_name['RideRequest']
_RIDEREPLY = DESCRIPTOR.message_types_by_name['RideReply']
_RIDERATING = DESCRIPTOR.message_types_by_name['RideRating']
_NULL = DESCRIPTOR.message_types_by_name['Null']
RideRequest = _reflection.GeneratedProtocolMessageType('RideRequest', (_message.Message,), {
  'DESCRIPTOR' : _RIDEREQUEST,
  '__module__' : 'joyride_pb2'
  # @@protoc_insertion_point(class_scope:joyride.RideRequest)
  })
_sym_db.RegisterMessage(RideRequest)

RideReply = _reflection.GeneratedProtocolMessageType('RideReply', (_message.Message,), {
  'DESCRIPTOR' : _RIDEREPLY,
  '__module__' : 'joyride_pb2'
  # @@protoc_insertion_point(class_scope:joyride.RideReply)
  })
_sym_db.RegisterMessage(RideReply)

RideRating = _reflection.GeneratedProtocolMessageType('RideRating', (_message.Message,), {
  'DESCRIPTOR' : _RIDERATING,
  '__module__' : 'joyride_pb2'
  # @@protoc_insertion_point(class_scope:joyride.RideRating)
  })
_sym_db.RegisterMessage(RideRating)

Null = _reflection.GeneratedProtocolMessageType('Null', (_message.Message,), {
  'DESCRIPTOR' : _NULL,
  '__module__' : 'joyride_pb2'
  # @@protoc_insertion_point(class_scope:joyride.Null)
  })
_sym_db.RegisterMessage(Null)

_JOYRIDE = DESCRIPTOR.services_by_name['JoyRide']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\030io.grpc.examples.joyrideB\014JoyrideProtoP\001\242\002\003HLW'
  _RIDEREQUEST._serialized_start=26
  _RIDEREQUEST._serialized_end=81
  _RIDEREPLY._serialized_start=83
  _RIDEREPLY._serialized_end=125
  _RIDERATING._serialized_start=127
  _RIDERATING._serialized_end=169
  _NULL._serialized_start=171
  _NULL._serialized_end=177
  _JOYRIDE._serialized_start=179
  _JOYRIDE._serialized_end=303
# @@protoc_insertion_point(module_scope)
