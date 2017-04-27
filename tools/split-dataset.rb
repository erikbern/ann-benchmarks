require 'set'

if ARGV.length != 5
    puts "Parameters: <inputfile> <outputdatafile> <outputqueryfile> <numberofqueries> <seed>"
    exit(-1)
end

inputfile = ARGV[0]
outputdatafile = ARGV[1]
outputqueryfile = ARGV[2]
k = ARGV[3].to_i
seed = ARGV[4].to_i

numOfPoints = File.foreach(inputfile).inject(0) {|c, line| c + 1}

puts "Found #{numOfPoints} points."
puts "Selecting #{k} of them at random."

a =* (0..(numOfPoints-1))
a.shuffle(random: Random.new(seed))
queryindices = a[0..(k - 1)].to_set

datafile = File.open(outputdatafile, "w")
queryfile = File.open(outputqueryfile, "w")

i = 0
File.open(inputfile).each_line do |line|
    if queryindices.include? i
        queryfile.puts line
    else
        datafile.puts line
    end
    i += 1
end

queryfile.close
datafile.close
