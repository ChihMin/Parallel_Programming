#!/usr/bin/env ruby 

if ARGV.size < 1
  puts "Please enter correct paramaters "
  exit 1
end


time_10 = Array.new
i = 0 
File.open(ARGV[0]).each_line do |line|
  if i % 48 == 0
  puts ""
    puts "| nodes | ppn | num of testcase | exe. time(s)|"
    puts "| :---: |:---: | :--------: | :-------------: |"
  end 

  arr = line.split(' ')
  node = arr[0].to_i
  ppn = arr[1].to_i
  n = arr[2].to_i
  minutes = arr[3].split('m')[0].to_f
  seconds = arr[3].split('m')[1].split('s')[0].to_f

  total = minutes * 60 + seconds
  time_10.push(total)
  puts "|#{node} | #{ppn} | #{n} | #{total} |" 
  
  i = i + 1
end
