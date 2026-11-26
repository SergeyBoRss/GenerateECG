return_code=0

echo "isort"
isort ecg ecg_noise.py random_ecg.py
return_code=$(($return_code | $?))

exit $return_code
