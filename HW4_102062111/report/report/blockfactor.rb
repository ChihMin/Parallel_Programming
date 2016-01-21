#!/usr/bin/env ruby

prefix = 'HW4_102062111'
apis = ['cuda', 'openmp']

apis.each do |api|
  name = prefix + "_#{api}"
  target = name + "_block.log"
  f = File.open(target, 'w')
  for i in 5..10
    block = 2 ** i
    command = "nvprof --metrics inst_integer ./#{name} testcase/1024.txt ans.txt #{block} 2>&1 | grep -i inst_integer"
    puts command
    ret = `#{command}`
    
    total_time = 0
    ret.split("\n").each do |line|
      time = line.split(" ")[-1].to_i
      total_time = total_time + time
    end 
    puts "#{name} #{block} #{total_time}"
    f.write("#{block} #{total_time}\n") 
  end
  f.close
end
