#!/bin/bash

function clean() {
  rm CHIHMIN* 
}
function clean_all() {
  rm CHIHMIN* new_job*
}

function get_time() {
  echo $2
}

function run_job() {

  if [ $# -lt 3 ]; then
    echo "Enter three argument : N & node & ppn "
    return 1
  fi
  
  src=basic_bcast
  exe=./$src
  report=${src}_run_report.txt
  job_name=CHIHMIN_JOB
  job=new_job_${1}_${2}_${3}.sh
  N=$1
  nodes=$2
  ppn=$3
  time_limit=3

  mpic++ ${src}.cpp -o $src
  if [ $? != 0 ]; then
    echo "Compile error, process is stopping ...."
    exit 1
  fi

  #rm $job
  echo "#PBS -q batch" >> $job
  echo "#PBS -N $job_name" >> $job
  echo "#PBS -r n" >> $job
  echo "#PBS -l nodes=$nodes:ppn=$ppn" >> $job
  echo "#PBS -l walltime=00:$time_limit:00" >> $job
  echo "cd \$PBS_O_WORKDIR" >> $job
  echo "time mpiexec $exe $N IN_84000000 output/testcase_${nodes}_${ppn}_${N}.out" >> $job 

  ret=$(qsub $job)
  echo "ret = $ret"

  file_exists=0
  while [ $file_exists != 1  ]; do
    pid=$(echo $ret | sed 's/.pp11//g')
    file_out=${job_name}.o${pid}
    file_err=${job_name}.e${pid}

    test -e ${job_name}.o${pid}
    if [ $? -eq 0 ]; then
      echo "Find $file, checking ret is error or not ... "
      err=$(cat $file_err | grep "execvp error")
      if [ ${#err} -eq 0 ]; then
        real_time=$(cat $file_err | grep real)
        file_exists=1
        echo "$nodes $ppn $N $(get_time $real_time)" >> $report
        echo "[SUCCESS] $nodes $ppn $N $(get_time $real_time)" 
      else
        clean
        ret=$(qsub $job)
        echo "Submit fail, resubmitting ...., ret = $ret"
      fi
    fi
    sleep 2s
  done
}

function go_bench() {
  i=0
  while [ $i -lt 3 ]; do
    i=$(($i+1))
    j=0
    while [ $j -lt 12 ]; do
      j=$(($j+1))
      run_job 100000 $i $j
    done
  done
}

mkdir -p output
clean_all
go_bench
