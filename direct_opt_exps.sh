#! /bin/bash

for c in ${1}; do
	echo "Runing experiment defined in ${c}"
	set -x
	python direct_opt/global_sh.py -c ${c}
	set +x
done

echo "Done running experiemnts in direct_opt."
