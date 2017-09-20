echo "Running pylint..."
echo
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
. $SCRIPTS_DIR/setup.sh
pylint --rcfile=$SCRIPTS_DIR/../pylintrc pypuffin

echo "...done."
echo

# Static type checker
# TODO - after some initial testing it seems like this isn't really quite ready for general use. It
# pretty much doesn't work at all with numpy/scipy/sklearn. If & when that changes it might be worth reconsidering

# echo "Running mypy..."
# echo
# mypy --ignore-missing-imports -p pypuffin
# echo "...done."
