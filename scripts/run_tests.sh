echo "Running all tests..."
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh
nosetests pypuffin --with-doctest -v
