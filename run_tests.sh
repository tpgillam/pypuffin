echo "Running doctests..."
for filename in $(find pypuffin -name *.py); 
do 
    echo "  $filename"
    python -m doctest $filename;
done

echo;
echo "Running unittests..."
for filename in $(find pypuffin -name *.py); 
do 
    echo "  $filename"
    python -m unittest $filename;
done

