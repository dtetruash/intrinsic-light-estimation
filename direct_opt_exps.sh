#! /bin/bash

for c in direct_opt/configs/config_*.ini; do
	echo "Runing experiment defined in ${c}"
	set -x
	python direct_opt/global_sh.py -c ${c}
	set +x
done

echo "Done running experiemnts in direct_opt."
