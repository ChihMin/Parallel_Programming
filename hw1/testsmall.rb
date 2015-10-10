#!/usr/bin/env ruby

require 'bindata'

n = 30
File.open("small.in", "wb") do |io|
  for i in 1..n
    num = (rand()*(10**10)).to_i % (2 * 10)
    sign = (rand()*(10**10)).to_i % 2
    if sign == 1
      num = num * -1
    end
    obj = BinData::Int32le.new(num)
    obj.write(io)
  end
end

puts "#{n}"
