require "rubygems"
require "unicode/scripts"
require "unicode/categories"

bool_to_si = -> (b) {
  b ? "1" : "0"
}

is_digit = -> (props) {
  (props & ["Nd", "No", "Nl"]).count > 0
}

is_letter = -> (props) {
  (props & ["LC", "Ll", "Lm", "Lo", "Lt", "Lu"]).count > 0
}

is_alpha = -> (props) {
  is_letter.call(props)
}

is_lower = -> (props) {
  (props & ["Ll"]).count > 0
}

is_upper = -> (props) {
  (props & ["Lu"]).count > 0
}

is_punct = -> (props) {
  (props & ["Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps"]).count > 0
}

if ARGV.length < 1
  $stderr.puts "Usage: ruby ./extract_unicharset.rb path/to/all-boxes > path/to/unicharset"
  exit
end

if !File.exist?(ARGV[0])
  $stderr.puts "The all-boxes file #{ARGV[0]} doesn't exist"
  exit
end

uniqs = IO.readlines(ARGV[0]).map { |line| line[0] }.uniq.sort

outs = uniqs.each_with_index.map do |char, ix|
  script = Unicode::Scripts.scripts(char).first
  props = Unicode::Categories.categories(char)

  isalpha = is_alpha.call(props)
  islower = is_lower.call(props)
  isupper = is_upper.call(props)
  isdigit = is_digit.call(props)
  ispunctuation = is_punct.call(props)

  props = [ isalpha, islower, isupper, isdigit, ispunctuation].reverse.inject("") do |state, is|
    "#{state}#{bool_to_si.call(is)}"
  end

  "#{char} #{props.to_i(2)} #{script} #{ix + 1}"
end

puts outs.count + 1
puts "NULL 0 Common 0"
outs.each { |o| puts o }
