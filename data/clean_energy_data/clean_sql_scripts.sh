#!/bin/bash

# Check if a file name was provided
if [ -z "$1" ]; then
  echo "Provide a filename"
  exit 1
fi

# Define the pattern to match
pattern='/*!40000 ALTER TABLE \`auth_user_user_permissions\` DISABLE KEYS */;'

# Use sed to comment out the matching lines in the provided file
sed -e "s#^$pattern#-- &#" "$1"

echo "Commented out occurrences of the pattern in $1."
