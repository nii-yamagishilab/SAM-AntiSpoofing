#!/bin/bash

mkdir wav

ls flac | parallel sox flac/{/.}.flac wav/{/.}.wav
