. ~/scripts/anaconda_setup.sh
SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export THEANO_FLAGS='device=cpu,force_device=True'
export KERAS_BACKEND=tensorflow
# Only add the path if it isn't already on the python path
if [[ $PYTHONPATH != *"pypuffin"* ]];
then
    export PYTHONPATH=$SCRIPTS_DIR/../:$PYTHONPATH
fi
