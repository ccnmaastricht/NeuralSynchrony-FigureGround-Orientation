#!/bin/bash

# Read in config file
config_file="../config/data/download.toml"
eval "$(toml query < "$config_file")"

# Create a directory to store the data files
mkdir -p ../../data

# Loop through the data files and download them
for file in "${files[@]}"; do
  url="$file.url"
  checksum="$file.checksum"
  filename="$(basename $url)"

  # Download the file
  curl -L -o "../../data/$filename" "$url"

  # Check the file integrity and remove if it fails
    if [[ "$(md5sum "../../data/$filename" | cut -d ' ' -f 1)" != "$checksum" ]]; then
      rm "../../data/$filename"
    fi

done
