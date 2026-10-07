# Usage: cmake -DINPUT_FILE=x.spv -DOUTPUT_FILE=x.spv.h -DVAR_NAME=name -P bin2c.cmake
# Emits: static const uint32_t NAME[] = {...}; static const size_t NAME_size = sizeof(NAME);
file(READ "${INPUT_FILE}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])"
       "0x\\4\\3\\2\\1," words "${hex}")
file(WRITE "${OUTPUT_FILE}"
"#pragma once
#include <stdint.h>
#include <stddef.h>
static const uint32_t ${VAR_NAME}[] = {${words}};
static const size_t ${VAR_NAME}_size = sizeof(${VAR_NAME});
")
