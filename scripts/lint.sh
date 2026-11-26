return_code=0

echo "flake8"
flake8 ecg ecg_noise.py random_ecg.py
return_code=$(($return_code | $?))

echo "mypy"
mypy ecg ecg_noise.py random_ecg.py
return_code=$(($return_code | $?))

echo "isort"
isort ecg ecg_noise.py random_ecg.py --diff --check-only
return_code=$(($return_code | $?))

exit $return_code
