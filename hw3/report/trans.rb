#!/usr/bin/env ruby


apis = ['MPI', 'OpenMP', 'Hybrid']
# apis = ['Hybrid']
ways = ['static', 'dynamic']
# testcase = [1000, 5000, 10000]

apis.each do |api|
  ways.each do |way|
  #testcase.each do |number|
    dir = 'balance'
    filename = "#{dir}/MS_#{api}_#{way}.txt"
    newFileName = "#{dir}/MS_#{api}_#{way}_new.txt"

    puts filename
    f = File.open(filename, 'r')
    w = File.open(newFileName, 'w')
    
    f.each do |str|
      time = str.split(' ')[-1]
      sec = time.split('m')[0].to_f * 60 +
              time.split('m')[1].split('s')[0].to_f
      puts sec
      
      list = str.split(' ')
      for i in 0..(list.size - 2)
        w.write("#{list[i]} ")
      end
      w.write("#{sec}\n")
    end

    f.close
    w.close
    
    `cp #{newFileName} #{filename}`
  #end
  end
end


