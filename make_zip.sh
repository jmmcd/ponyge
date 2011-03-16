#! /bin/sh

# Script for making a zip file of ponyge and tagging the release
# e.g. sh ./make_zip.sh 0.1.1 creates the file ponyge-0.1.1.zip
# sh ./make_zip.sh 0.1.1 TAG creates the file ponyge-0.1.1.zip and copies the trunk to tags as release-0.1.1

EXPECTED_ARGS=1
E_BADARGS=65
TAG="TAG"
USERNAME=erik.hemberg
USERNAME=jamesmichaelmcdermott

if [ $# -lt $EXPECTED_ARGS ]
then
  echo "Usage: `basename $0` version_number [$TAG]"
  exit $E_BADARGS
fi

NAME="ponyge-$1"

# Exporting the trunk to a zip
svn export https://ponyge.googlecode.com/svn/trunk/ --username $USERNAME $NAME

zip -r $NAME.zip $NAME

if [ "$2" = $TAG ]; then
# Tagging the release by copying the trunk to tags
    svn copy https://ponyge.googlecode.com/svn/trunk/ https://ponyge.googlecode.com/svn/tags/$NAME --username $USERNAME
fi