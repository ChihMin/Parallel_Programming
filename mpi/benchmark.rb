#!/usr/bin/env ruby

list = Array.new
for i in 1..40
  ret = `./run.sh #{i} 100000 testcase.in testcase.out 2>&1 | grep -i real`
  puts "#{i} --> #{ret}"
end 
