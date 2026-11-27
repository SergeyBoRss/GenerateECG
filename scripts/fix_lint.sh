return_code=0

echo "isort"
isort ecg random_ecg.py
return_code=$(($return_code | $?))

exit $return_code
