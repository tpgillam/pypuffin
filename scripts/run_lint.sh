echo "Running pylint..."
echo
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh
pylint --rcfile=$SCRIPTS_DIR/../pylintrc pypuffin
