#!/usr/bin/env ruby

require 'bindata'

n = 2**31 -1
File.open("test.in", "wb") do |io|
  for i in 1..n
    num = (rand()*(10**100)).to_i % (2 * 10**9)
    sign = (rand()*(10**16)).to_i % 2
    if sign == 1
      num = num * -1
    end
    obj = BinData::Int32le.new(num)
    obj.write(io)
  end
end

puts "#{n}"
