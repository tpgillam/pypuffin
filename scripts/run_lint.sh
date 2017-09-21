SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh

if [ $# -eq 0 ]; then
    # No argument supplied - default to running lint on everything
    module="pypuffin"
else
    module=$1
fi

echo "Running pylint for $module..."
pylint --rcfile=$SCRIPTS_DIR/../pylintrc $module
echo "...done."

# Static type checker
# TODO - after some initial testing it seems like this isn't really quite ready for general use. It
# pretty much doesn't work at all with numpy/scipy/sklearn. If & when that changes it might be worth reconsidering

# echo "Running mypy..."
# echo
# mypy --ignore-missing-imports -p pypuffin
# echo "...done."
