SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh

if [ $# -eq 0 ]; then
    # No argument supplied - default to running lint on everything
    module="pypuffin"
else
    module=$1
fi

echo "Running tests for $module..."
python -m pypuffin.app.unittest $module
echo "...done."
