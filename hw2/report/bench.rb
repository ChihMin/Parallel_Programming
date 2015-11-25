#!/usr/bin/env ruby

class Bench
  def initialize(execName = nil)
    @exec = execName
    @fileName = @exec + ".txt"
  end

  def run_SRCC
    @file = File.open(@fileName, "w")
    for i in 1..10
      puts "#{i} / 10"
      for j in 0..4
        capacity = i
        carTime = 10**j
        puts carTime
        ret = `./#{@exec} 10 #{capacity} #{carTime} 1000 | grep -i "Average waiting"`
        times = ret.split(' ')[4].to_f
        @file.write("#{capacity} #{carTime} #{times}\n")
      end
    end
    @file.close
  end

  def run
    if @exec == "hw2_SRCC"
      run_SRCC
    elsif @exec == "hw2_SRCC_long"
      run_SRCC
    end
  end
end

if ARGV.size() < 1
  puts "error arguments ... "
  exit 1 
end

bench = Bench.new(ARGV[0])
bench.run

