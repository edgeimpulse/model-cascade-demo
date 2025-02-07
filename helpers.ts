import sharp from 'sharp';
import { ModelInformation, RunnerClassifyResponseSuccess } from './edge-impulse-linux/library/classifier/linux-impulse-runner';

export function getBoundingBoxesFromGrid(grid: number[][]): { x: number, y: number, w: number, h: number }[] {
    const visited: boolean[][] = grid.map(row => row.map(() => false));
    const boundingBoxes: { x: number, y: number, w: number, h: number }[] = [];

    // Directions for 8-way traversal (up, down, left, right, and diagonals)
    const directions = [
        [-1, 0], [1, 0], [0, -1], [0, 1],      // Up, down, left, right
        [-1, -1], [-1, 1], [1, -1], [1, 1]     // Diagonals: top-left, top-right, bottom-left, bottom-right
    ];

    function isValid(x: number, y: number): boolean {
        return x >= 0 && x < grid[0].length && y >= 0 && y < grid.length;
    }

    function bfs(startX: number, startY: number) {
        let minX = startX, maxX = startX, minY = startY, maxY = startY;
        const queue: [number, number][] = [[startX, startY]];
        visited[startY][startX] = true;

        while (queue.length > 0) {
            const [x, y] = queue.shift()!;

            // Update bounding box coordinates
            minX = Math.min(minX, x);
            maxX = Math.max(maxX, x);
            minY = Math.min(minY, y);
            maxY = Math.max(maxY, y);

            // Traverse neighboring cells (including diagonals)
            for (const [dx, dy] of directions) {
                const newX = x + dx;
                const newY = y + dy;

                if (isValid(newX, newY) && grid[newY][newX] === 1 && !visited[newY][newX]) {
                    visited[newY][newX] = true;
                    queue.push([newX, newY]);
                }
            }
        }

        // Calculate width and height of the bounding box
        const w = maxX - minX + 1;
        const h = maxY - minY + 1;
        boundingBoxes.push({ x: minX, y: minY, w, h });
    }

    // Iterate over the grid to find all disconnected groups of 1s
    for (let y = 0; y < grid.length; y++) {
        for (let x = 0; x < grid[y].length; x++) {
            if (grid[y][x] === 1 && !visited[y][x]) {
                bfs(x, y);
            }
        }
    }

    return boundingBoxes;
}

export async function highlightAnomalyInImage(img: Buffer, ev: RunnerClassifyResponseSuccess, model: ModelInformation) {
    let imgMetadata = await sharp(img).metadata();

    // first need to resize the img to same shape os modelParameters
    let lowestFactor = Math.min(
        imgMetadata.width! / model.modelParameters.image_input_width,
        imgMetadata.height! / model.modelParameters.image_input_height
    );
    let resizedWidth = Math.round(model.modelParameters.image_input_width * lowestFactor);
    let resizedHeight = Math.round(model.modelParameters.image_input_height * lowestFactor);
    let resizedImg = await sharp(img).resize({
        width: resizedWidth,
        height: resizedHeight,
    });

    if (ev.result.visual_anomaly_grid && ev.result.visual_anomaly_grid.length > 0) {
        let cellWidth = ev.result.visual_anomaly_grid[0].width;
        let cellHeight = ev.result.visual_anomaly_grid[0].height;

        // make grid like [ [ 0, 0 ], [ 0, 0 ] ] => fill each cell thats anomalous with a 1
        let grid: number[][] = Array.from({ length: model.modelParameters.image_input_height / cellHeight },
            () => Array(model.modelParameters.image_input_width / cellWidth).fill(0));
        for (let cell of ev.result.visual_anomaly_grid) {
            for (let cellX = cell.x; cellX < cell.x + cell.width; cellX += cellWidth) {
                for (let cellY = cell.y; cellY < cell.y + cell.height; cellY += cellHeight) {
                    let x = cellX / cellWidth;
                    let y = cellY / cellHeight;
                    grid[y][x] = 1;
                }
            }
        }

        let anomalyBbs = getBoundingBoxesFromGrid(grid).map(x => {
            return {
                x: x.x * cellWidth,
                y: x.y * cellHeight,
                w: x.w * cellWidth,
                h: x.h * cellHeight
            };
        });

        let strokeWidth = Math.floor(2 * lowestFactor);
        let strokeOffset = Math.ceil(1 * lowestFactor);

        let overlayRects: { x: number, y: number, w: number, h: number }[] = [];
        for (let bb of anomalyBbs) {
            let x = (bb.x - strokeWidth) * lowestFactor;
            let y = (bb.y - strokeWidth) * lowestFactor;
            let w = (bb.w + (strokeWidth * 2)) * lowestFactor;
            let h = (bb.h + (strokeWidth * 2)) * lowestFactor;
            if (x < strokeOffset) { x = strokeOffset; }
            if (y < strokeOffset) { y = strokeOffset; }
            if (x + w + strokeOffset > resizedWidth) {
                w = resizedWidth - x - strokeOffset;
            }
            if (y + h + strokeOffset > resizedHeight) {
                h = resizedHeight - y - strokeOffset;
            }
            overlayRects.push({ x, y, w, h });
        }

        const svgOverlay = `
            <svg width="${model.modelParameters.image_input_width * lowestFactor}" height="${model.modelParameters.image_input_height * lowestFactor}">
                ${overlayRects.map(bb => `
                <rect x="${bb.x}" y="${bb.y}"
                    width="${bb.w}" height="${bb.h}"
                    fill="rgba(0,0,0,0)" stroke="rgba(255, 0, 0, 0.5)" stroke-width="${strokeWidth}" />
                `)}
            </svg>
            `;

        // Read and process the image
        return await resizedImg
            .composite([{ input: Buffer.from(svgOverlay), blend: 'over' }]) // Overlay the rectangle
            .png()
            .toBuffer();
    }

    return img;
}
