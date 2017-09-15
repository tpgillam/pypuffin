echo "Running all tests..."
echo
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh
python -m pypuffin.app.unittest
