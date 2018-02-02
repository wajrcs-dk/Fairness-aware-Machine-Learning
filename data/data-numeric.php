<?php
$data = array();
$i = array();
// $file = 'census-income/census-income.numeric.data';
$file = 'german-credit/german-credit.numeric.data';
file_put_contents($file, '');

// $handle = fopen("census-income/census-income.data", "r");
$handle = fopen("german-credit/german-credit.data", "r");
if ($handle) {
    while (($line = fgets($handle)) !== false) {
        // process the line read.
        $line = explode(" ", $line);
        foreach ($line as $key => $value) {
        	$line = str_replace("\r", '', $line);

            if (!isset($i[$key])) {
                $i[$key] = 0;
            }

        	if (!is_numeric($value)) {
        		if (!isset($data[$key][$value])) {
                    $data[$key][$value] = $i[$key]++;
        		}
        		$line[$key] = $data[$key][$value];
        	} else {
        		$line[$key] = trim($value);
        	}

        }
        $line = implode(',', $line);
        file_put_contents($file, $line."\n", FILE_APPEND | LOCK_EX);
        // echo $line, PHP_EOL;

    }

    fclose($handle);
} else {
    // error opening the file.
}

print_r($data);

?>