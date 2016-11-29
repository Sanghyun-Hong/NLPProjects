use strict;
use warnings;

my $input_submission_file = shift @ARGV;

my $index_file = "multi.gs.mapping.lines.txt";

open(INDEX, "<$index_file") or die "Could not open index file\n";

my @lineIds = <INDEX>;
close INDEX;

open(SCORES, "<$input_submission_file") or die "Could not open $input_submission_file\n";
my @scores = <SCORES>;
close SCORES;

foreach my $lineID (@lineIds) {
    print $scores[$lineID];
}


