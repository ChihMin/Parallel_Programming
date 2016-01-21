#!/usr/bin/env ruby

if ARGV.size < 2
  puts "Please enter n/m "
  exit 1
end

f = File.open('testcase.txt', 'w')
N = ARGV[0].to_i
M = ARGV[1].to_i

f.write("#{N} #{M}\n")
for i in 1..M
    x = 2
    y = 2
    w = (rand() * 100).to_i + 1
    while x == y
      x = (rand() * N).to_i + 1
      y = (rand() * N).to_i + 1
    end
    f.write("#{x} #{y} #{w}\n")
end
f.close
