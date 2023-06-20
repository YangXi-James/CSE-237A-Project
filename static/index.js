// Define the chart data
const temperatureData = {
    labels: [],
    datasets: [{
        label: 'Temperature',
        data: [],
        borderColor: 'red',
        fill: false
    }]
};

const airQualityData = {
    labels: [],
    datasets: [{
        label: 'Air Quality',
        data: [],
        borderColor: 'green',
        fill: false
    }]
};

const headCountData = {
    labels: [],
    datasets: [{
        label: 'Head Count',
        data: [],
        borderColor: 'blue',
        fill: false
    }]
};

const speakingTimeData = {
    labels: [],
    datasets: [{
        label: 'Speaking Time',
        data: [],
        backgroundColor: [
            '#F1F7ED',
            '#243E36',
            '#7CA982',
            '#E0EEC6',
            '#C2A83E',
            '#C4E7D4',
            '#B9C0DA',
            '#63585E'
        ]
    }]
};

// Create the charts
const temperatureChart = new Chart('temperature-chart', {
    type: 'line',
    data: temperatureData
});

const airQualityChart = new Chart('air-quality-chart', {
    type: 'line',
    data: airQualityData
});

const headCountChart = new Chart('head-count-chart', {
    type: 'line',
    data: headCountData
});

const speakingTimeChart = new Chart('speaking-time-chart', {
    type: 'pie',
    data: speakingTimeData
});

// Periodically fetch data from the server and update the charts
const temp_aq_fetch = setInterval(() => {
    // Fetch temperature data
    fetch('/temperature')
        .then(response => response.json())
        .then(data => {
            console.debug(`Temp: ${data.temperature}`)
            temperatureData.labels.push(new Date().toLocaleTimeString());
            temperatureData.datasets[0].data.push(data.temperature);
            temperatureChart.update();
        });

    // Fetch air quality data
    fetch('/air_quality')
        .then(response => response.json())
        .then(data => {
            console.debug(`AQ: ${data.air_quality}`)
            airQualityData.labels.push(new Date().toLocaleTimeString());
            airQualityData.datasets[0].data.push(data.air_quality);
            airQualityChart.update();
        });
}, 1000); // Update every 5 seconds

const hc_fetch = setInterval(() => {    // Fetch head count data
    fetch('/head_count')
        .then(response => response.json())
        .then(data => {
            console.debug(`Head count: ${data.head_count}`)
            headCountData.labels.push(new Date().toLocaleTimeString());
            headCountData.datasets[0].data.push(data.head_count);
            headCountChart.update();
        });
}, 10000);

const sd_fetch = setInterval(() => {
    // Fetch speaking time data
    fetch('/speech_diarization')
        .then(response => response.json())
        .then(data => {
            speech_diarization = data.speech_diarization;

            console.log("Parsing Speech Diarization Results...")
            console.log(speech_diarization)
            for (const [speaker_id, duration] of Object.entries(speech_diarization)) {
                console.log(`${speaker_id}: ${duration}`);

                speaker_index = speakingTimeData.labels.findIndex(elem => elem === speaker_id);
                if (speaker_index === -1) {
                    // speaker is not find
                    speakingTimeData.labels.push(speaker_id);
                    speakingTimeData.datasets[0].data.push(+duration);
                } else {
                    // speaker is found
                    speakingTimeData.datasets[0].data[speaker_index] += +duration;
                }
            }

            speakingTimeChart.update();
        });
}, 10000)

document.getElementById("speaking-time-chart").style.height = document.getElementById("head-count-chart").style.height;

window.onbeforeunload = () => {
    clearInterval(temp_aq_fetch);
    clearInterval(hc_fetch);
    clearInterval(sd_fetch);

    console.log("Clean up complete.")
}
