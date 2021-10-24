#!/bin/bash
CUR_VER=$(git describe --tag)
printf "Current version: %s\n" $CUR_VER
git fetch --tags
LATEST_VER=$(git tag | tail -1)
printf "Latest version: %s\n" $LATEST_VER
printf "Update (y/n): "
read UPDATE
if [ "$UPDATE" = "y" ]; then
  printf "Updating...\n"
  git checkout tags/$LATEST_VER &> /dev/null
else
  printf "Not updating...\n"
fi