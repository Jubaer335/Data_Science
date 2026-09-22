function preprocessData(data, columns) {
    let logs = [];
    function addLog(msg) { logs.push(msg); }

    // 1. Drop Unnamed columns and empty columns
    let initialColCount = columns.length;
    columns = columns.filter(col => col && col.trim() !== '' && !col.includes('Unnamed:'));
    let droppedCols = initialColCount - columns.length;
    if (droppedCols > 0) addLog(`Dropped ${droppedCols} empty or 'Unnamed' column(s).`);

    // PapaParse might assign keys incorrectly if there are empty headers or trailing tabs.
    // The Python pandas snippet checks subset=['Course','Gender','Target'].
    let courseKey = 'Course';
    let genderKey = 'Gender';

    // PapaParse has misaligned the headers with the data because of an empty column.
    // Let's find the true target key if there are empty columns shifting things.
    let targetKey = 'Target';
    if (data.length > 0) {
        for (let col of columns) {
            let sampleVal = data.map(r => r[col]).find(v => v === 'Dropout' || v === 'Graduate' || v === 'Enrolled');
            if (sampleVal) {
                targetKey = col;
                break;
            }
        }
    }

    // 2. Drop rows with NaN in 'Target' (Using targetKey because of TSV shift)
    let initialRowCount = data.length;
    data = data.filter(row => {
        return row[targetKey] != null && String(row[targetKey]).trim() !== '';
    });
    let droppedRows = initialRowCount - data.length;
    if (droppedRows > 0) addLog(`Dropped ${droppedRows} row(s) missing values in Target.`);

    // Helper for imputation
    const imputeMode = (col) => {
        let counts = {};
        let maxCount = 0;
        let modeValue = null;
        data.forEach(row => {
            let val = row[col];
            if (val != null && val !== '') {
                counts[val] = (counts[val] || 0) + 1;
                if (counts[val] > maxCount) {
                    maxCount = counts[val];
                    modeValue = val;
                }
            }
        });
        let imputedCount = 0;
        data.forEach(row => {
            if (row[col] == null || row[col] === '') {
                row[col] = modeValue;
                imputedCount++;
            }
        });
        if (imputedCount > 0) addLog(`Imputed ${imputedCount} missing values in '${col}' with mode (${modeValue}).`);
    };

    const imputeMean = (col, floor = false) => {
        let sum = 0, count = 0;
        data.forEach(row => {
            let val = parseFloat(row[col]);
            if (!isNaN(val)) { sum += val; count++; }
        });
        let mean = sum / count;
        let fillValue = floor ? Math.floor(mean) : mean;

        let imputedCount = 0;
        data.forEach(row => {
            let val = parseFloat(row[col]);
            if (isNaN(val)) {
                row[col] = fillValue;
                imputedCount++;
            } else {
                row[col] = val; // ensure numeric
            }
        });
        if (imputedCount > 0) addLog(`Imputed ${imputedCount} missing values in '${col}' with ${floor ? 'floored ' : ''}mean (${fillValue.toFixed(2)}).`);
    };

    // Impute Most Frequent (Mode)
    const modeCols = [
        'Daytime/evening attendance', 'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Scholarship holder', 'International'
    ];
    modeCols.forEach(col => { if(columns.includes(col)) imputeMode(col); });

    // Impute Mean (Floored)
    const flooredMeanCols = [
        'Marital status', 'Application mode', 'Application order',
        'Previous qualification', 'Nacionality', "Mother's qualification",
        "Father's qualification", "Mother's occupation", "Father's occupation", "Age at enrollment"
    ];
    flooredMeanCols.forEach(col => { if(columns.includes(col)) imputeMean(col, true); });

    // Impute Mean (Standard)
    const meanCols = ['Previous qualification (grade)', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP'];
    // We shouldn't impute the column if it actually holds the Target string data due to a TSV shift!
    meanCols.forEach(col => { if(columns.includes(col) && col !== targetKey) imputeMean(col, false); });

    // Label Encoding for 'Target'
    let labels = Array.from(new Set(data.map(row => row[targetKey]).filter(t => t != null && String(t).trim() !== ''))).sort();
    if (labels.length > 0) {
        data.forEach(row => {
            row['Target_enc'] = labels.indexOf(row[targetKey]);
            // Re-assign target properly if it was misaligned
            if (targetKey !== 'Target') {
                row['Target'] = row[targetKey];
            }
        });
        if (!columns.includes('Target')) columns.push('Target');
        columns.push('Target_enc');
        addLog(`Encoded 'Target' column to 'Target_enc' (${labels.join(', ')} -> ${labels.map((_, i) => i).join(', ')}).`);
    } else {
        addLog("No valid 'Target' values found for encoding.");
    }

    return { data, columns, logs };
}
