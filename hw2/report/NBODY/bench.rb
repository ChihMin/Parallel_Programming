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
  
  def run_nbody
    
    execList = ["hw2_NB_pthread", 
                "hw2_NB_openmp", 
                "hw2_NB_BHalgo", 
                "hw2_NB_BHalgo_1"]

    tmpList = ["hw2_NB_BHalgo"]
    theta = ["0.1", "0.1", "0.1", "1.0"]
    
    execList.each do |exec|
      i = 0
      filename = exec + ".txt"
      @file = File.open(filename, "w")
      
      # For strong scalability
      target = "test4.txt"
      
      j = 1
      while j < 800
        steps = j
        params = "40 1 #{steps} 1 #{target} #{theta[i]} disable -1 -1 2.5 500"
        command =  "./run.sh ./#{exec} #{params} 2>&1 > /dev/null | grep real"
        puts command
        ret = `#{command}`
        
        runTime = ret.split(' ')[1]
        minutes = runTime.split('m')[0].to_f
        seconds = runTime.split('m')[1].split('s')[0].to_f + 60 * minutes
        puts "#{exec} #{j} #{seconds}"
        @file.write("#{exec} #{j} #{seconds}\n")
        j = j + 100 
      end
      i = i + 1
      @file.close
    end
    
  end
  
  def run
    if @exec == "hw2_SRCC"
      run_SRCC
    elsif @exec == "hw2_SRCC_long"
      run_SRCC
    elsif @exec == "nbody"
      run_nbody 
    end
  end
end

if ARGV.size() < 1
  puts "error arguments ... "
  exit 1 
end

bench = Bench.new(ARGV[0])
bench.run

