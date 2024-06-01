#!/bin/bash
clear_folders() {
  directory="${root_folder}/ma_llm/selfalign/results/aug"

  if [ -z "$directory" ]; then
    echo "Please provide a directory path."
    return 1
  fi

  if [ ! -d "$directory" ]; then
    echo "Directory '$directory' does not exist."
    return 1
  fi

  find "$directory" -type d -empty -delete
}
