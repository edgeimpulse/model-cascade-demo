import sharp from 'sharp';
import express = require('express');
import socketIO from 'socket.io';
import http from 'http';
import Path from 'path';
import OpenAI from "openai";
import { highlightAnomalyInImage } from "./helpers";
import { Ffmpeg, ICamera, ImageClassifier, Imagesnap, LinuxImpulseRunner, ModelInformation, RunnerHelloHasAnomaly } from './edge-impulse-linux/library';

if (!process.env.OPENAI_API_KEY) {
    console.log('Missing OPENAI_API_KEY');
    process.exit(1);
}

const BASE_PROMPT = `This image was flagged by an anomaly detection system, the anomaly is flagged in red. Can you explain what is off in this picture?

Reply with a very short response.`;

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    try  {

        // Required arguments:
        //   arg 2: Path to the model file. e.g. /tmp/model.eim
        //   arg 3: Name of the camera device, see output of `gst-device-monitor-1.0`. e.g. "HD Pro Webcam C920"
        // Optional arguments:
        //   arg 4: desired FPS. e.g. 20, default 5
        //   arg 5: desired capture width. e.g. 320, default 640
        //   arg 6: desired capture height. e.g. 200, default 480
        //   arg 7: webserver port. e.g. 4999, default 4912

        const argModelFile = process.argv[2];
        const argCamDevice = process.argv[3];
        const fps = process.argv[4] ? Number(process.argv[4]) : 5;
        const dimensions = (process.argv[5] && process.argv[6]) ? {
            width: Number(process.argv[5]),
            height: Number(process.argv[6])
        } : {
            width: 640,
            height: 480
        };

        const port = process.argv[7] ? Number(process.argv[7]) : (process.env.PORT ? Number(process.env.PORT) : 4912);

        if (!argModelFile) {
            console.log('Missing one argument (model file)');
            process.exit(1);
        }

        let runner = new LinuxImpulseRunner(argModelFile);
        let model = await runner.init();

        let labels = model.modelParameters.labels;
        if (model.modelParameters.has_anomaly === RunnerHelloHasAnomaly.VisualGMM) {
            labels.push('anomaly');
        }

        console.log('Starting the image classifier for',
            model.project.owner + ' / ' + model.project.name, '(v' + model.project.deploy_version + ')');
        console.log('Parameters',
            'image size', model.modelParameters.image_input_width + 'x' + model.modelParameters.image_input_height + ' px (' +
                model.modelParameters.image_channel_count + ' channels)',
            'classes', labels);

        // select a camera... you can implement this interface for other targets :-)
        let camera: ICamera;
        if (process.platform === 'darwin') {
            camera = new Imagesnap();
        }
        else if (process.platform === 'linux') {
            camera = new Ffmpeg(false /* verbose */);
        }
        else {
            throw new Error('Unsupported platform "' + process.platform + '"');
        }
        await camera.init();

        const devices = await camera.listDevices();
        if (devices.length === 0) {
            throw new Error('Cannot find any webcams');
        }
        if (devices.length > 1 && !argCamDevice) {
            throw new Error('Multiple cameras found (' + devices.map(n => '"' + n + '"').join(', ') + '), add ' +
                'the camera to use to this script (node classify-camera-webserver.js model.eim cameraname)');
        }

        let device = argCamDevice || devices[0];

        console.log('Using camera', device, 'starting...');

        await camera.start({
            device: device,
            intervalMs: 1000 / fps,
            dimensions: dimensions
        });

        camera.on('error', error => {
            console.log('camera error', error);
            process.exit(1);
        });

        console.log('Connected to camera');

        let imageClassifier = new ImageClassifier(runner, camera);

        await imageClassifier.start();

        let webserverPort = await startWebServer(model, camera, imageClassifier, port);
        console.log('');
        console.log('Want to see a feed of the camera and live classification in your browser? ' +
            'Go to http://localhost:' + webserverPort);
        console.log('');
    }
    catch (ex) {
        console.error(ex);
        process.exit(1);
    }
})();

function startWebServer(model: ModelInformation, camera: ICamera, imgClassifier: ImageClassifier, port: number) {
    let openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

    const app = express();
    app.use(express.static(Path.join(__dirname, '..', 'public')));

    const server = new http.Server(app);
    const io = socketIO(server);

    let cascadeEnabled = false;
    let prompt = BASE_PROMPT;
    let thresholdObj = <{ id: number, min_anomaly_score: number } | undefined>
        (model.modelParameters.thresholds || []).find(x => x.type === 'anomaly_gmm');

    // you can also get the actual image being classified from 'imageClassifier.on("result")',
    // but then you're limited by the inference speed.
    // here we get a direct feed from the camera so we guarantee the fps that we set earlier.

    let lastFrameOriginalResolution: Buffer | undefined;
    let nextFrame = Date.now();
    let processingFrame = false;
    camera.on('snapshot', async (data) => {
        if (nextFrame > Date.now() || processingFrame) return;

        lastFrameOriginalResolution = data;

        processingFrame = true;

        let img;
        if (model.modelParameters.image_channel_count === 3) {
            img = sharp(data).resize({
                height: model.modelParameters.image_input_height,
                width: model.modelParameters.image_input_width
            });
        }
        else {
            img = sharp(data).resize({
                height: model.modelParameters.image_input_height,
                width: model.modelParameters.image_input_width
            }).toColourspace('b-w');
        }

        io.emit('image', {
            img: 'data:image/jpeg;base64,' + (await img.jpeg().toBuffer()).toString('base64')
        });

        nextFrame = Date.now() + 50;
        processingFrame = false;
    });

    let isRunningOpenAI = false;

    let imgToSendToOpenAI: Buffer = Buffer.from([ ]);
    let anomalySeen = false;

    (async () => {
        const sleep = (ms: number) => new Promise<void>(res => setTimeout(res, ms));
        while (1) {
            await sleep(500);

            if (!anomalySeen) continue;
            if (!cascadeEnabled) continue;
            if (!prompt) continue;

            // wait 500ms. to go into focus
            await sleep(500);
            if (!anomalySeen) continue;

            try {
                console.log('Anomaly detected, asking GPT-4o...');
                io.emit('anomaly', {
                    message: 'Anomaly detected, asking GPT-4o...',
                });

                const resp = await openai.chat.completions.create({
                    model: 'gpt-4o-2024-05-13',
                    messages: [{
                        role: 'user',
                        content: [{
                            type: 'text',
                            text: prompt,
                        }, {
                            type: 'image_url',
                            image_url: {
                                url: 'data:image/jpeg;base64,' + (imgToSendToOpenAI.toString('base64')),
                                detail: 'auto'
                            }
                        }]
                    }]
                });

                if (resp.choices.length !== 1) {
                    throw new Error('Expected choices to have 1 item (' + JSON.stringify(resp) + ')');
                }
                if (resp.choices[0].message.role !== 'assistant') {
                    throw new Error('Expected choices[0].message.role to equal "assistant" (' + JSON.stringify(resp) + ')');
                }
                if (typeof resp.choices[0].message.content !== 'string') {
                    throw new Error('Expected choices[0].message.content to be a string (' + JSON.stringify(resp) + ')');
                }

                console.log('Response:', resp.choices[0].message.content);
                console.log('');

                io.emit('anomaly', {
                    message: 'Response: ' + resp.choices[0].message.content,
                });
            }
            catch (ex) {
                console.log('OpenAI failed:', ex);
                console.log('');
            }
        }
    })();

    imgClassifier.on('result', async (ev, timeMs, imgAsJpg) => {
        io.emit('classification', {
            modelType: model.modelParameters.model_type,
            result: ev.result,
            timeMs: timeMs,
            additionalInfo: ev.info,
        });

        if (lastFrameOriginalResolution) {
            imgToSendToOpenAI = await highlightAnomalyInImage(lastFrameOriginalResolution, ev, model);

            if (ev.result.visual_anomaly_grid && ev.result.visual_anomaly_grid.length > 0) {
                await sharp(imgToSendToOpenAI).toFile('tmp.png');
            }
        }

        if ((ev.result.visual_anomaly_grid || []).length > 0 && !isRunningOpenAI) {
            anomalySeen = true;
        }
        else {
            anomalySeen = false;
        }
    });

    io.on('connection', socket => {
        socket.emit('hello', {
            projectName: model.project.owner + ' / ' + model.project.name,
            canSetThreshold: !!thresholdObj,
            defaultThreshold: thresholdObj?.min_anomaly_score,
        });

        socket.on('cascade-enable', () => {
            cascadeEnabled = true;
        });
        socket.on('cascade-disable', () => {
            cascadeEnabled = false;
        });

        socket.on('prompt', (promptArg: string) => {
            prompt = promptArg;
        });

        socket.on('threshold-override', async (thresholdOverrideArg: string) => {
            try {
                if (thresholdOverrideArg && !isNaN(Number(thresholdOverrideArg)) && thresholdObj) {
                    let v = Number(thresholdOverrideArg);
                    if (v === thresholdObj.min_anomaly_score) return;

                    console.log(`Updating threshold, now: ${thresholdObj.min_anomaly_score}, setting to: ${v}...`);

                    await imgClassifier.getRunner().setLearnBlockThreshold({
                        id: thresholdObj.id,
                        type: 'anomaly_gmm',
                        min_anomaly_score: Number(thresholdOverrideArg),
                    });

                    thresholdObj.min_anomaly_score = Number(thresholdOverrideArg);

                    console.log(`Updated threshold to ${v}`);
                }
            }
            catch (ex) {
                console.log('Failed to set threshold:', ex);
            }
        });
    });

    return new Promise<number>((resolve) => {
        server.listen(port, process.env.HOST || '0.0.0.0', async () => {
            resolve(port);
        });
    });
}
