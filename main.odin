package main

import "core:encoding/endian"
import "core:encoding/csv"
import "core:fmt"
import "core:os"
import "core:math/rand"
import "core:simd"


IMAGE_SIZE :: 784

MnistImageHeader :: struct #packed 
{
    magicNumber: u32,
    count: u32,
    rows: u32,
    cols: u32,
}

MnistLabelHeader :: struct #packed
{
    magicNumber: u32,
    labelsNum: u32,
}

MnistData :: struct
{
    images: [dynamic]f32,
    labels: [dynamic]u8,
    imageHeader: MnistImageHeader,
    labelHeader: MnistLabelHeader,
}

Adaline :: struct 
{
    weights: [IMAGE_SIZE + 1]f32,
    eta: f32,
    label: int,
}

ReadMnistFile :: proc(data: ^MnistData, imageFileName: string, labelFileName: string) -> (ok: bool)
{
    imgBytes, err := os.read_entire_file_from_path(imageFileName, context.allocator)

    if err != nil
    {   
        fmt.eprintfln("Failed to load the file: %v", err)
        return false 
    }
    defer delete(imgBytes)

    data.imageHeader.magicNumber = endian.get_u32(imgBytes[0:], .Big) or_return
    data.imageHeader.count = endian.get_u32(imgBytes[4:], .Big) or_return
    data.imageHeader.rows = endian.get_u32(imgBytes[8:], .Big) or_return
    data.imageHeader.cols = endian.get_u32(imgBytes[12:], .Big) or_return

    if data.imageHeader.magicNumber != 2051
    {
        return false
    }

    labelBytes, err2 := os.read_entire_file_from_path(labelFileName, context.allocator)
    
    if err2 != nil
    {
        fmt.eprintfln("Failed to load the file: %v", err2)
        return false 
    }
    defer delete(labelBytes)
    
    data.labelHeader.magicNumber = endian.get_u32(imgBytes[0:], .Big) or_return
    data.labelHeader.labelsNum = endian.get_u32(labelBytes[4:], .Big) or_return

    totalPixels := data.imageHeader.count * data.imageHeader.rows * data.imageHeader.cols
    //data.images = make([dynamic]u8, data.imageHeader.count * data.imageHeader.rows * data.imageHeader.cols * size_of(u8))
    resize(&data.images, totalPixels)
    inv255 := f32(1.0) / 255.0

    for i in 0..<totalPixels
    {
        data.images[i] = f32(imgBytes[16 + i]) * inv255
    }

    //data.labels = make([dynamic]u8, data.labelHeader.labelsNum * size_of(u8))
    resize(&data.labels, data.labelHeader.labelsNum * size_of(u8))
    copy(data.labels[:], labelBytes[8:])

    return true
}

InitAdalineUnit :: proc(adaline: ^Adaline, num: int)
{
    adaline.weights[0] = 1
    for i := 1; i < len(adaline.weights); i+=1
    {
        adaline.weights[i] = rand.float32_range(-0.1, 0.1)
    }

    adaline^.eta = 0.001
    adaline^.label = num
}

GetImage :: proc(trainData: ^MnistData, index: int) -> []f32
{
    offset := index * IMAGE_SIZE
    return trainData.images[offset : offset + IMAGE_SIZE]
}

DotProduct :: proc(weights: []f32, input: []f32) -> f32
{
    sumVec: simd.f32x16 = 0.0

    for i := 0; i < len(weights); i+=16
    {
        w := simd.from_slice(simd.f32x16, weights[i:])
        x := simd.from_slice(simd.f32x16, input[i:])

        sumVec += w * x
    }

    return simd.reduce_add_ordered(sumVec)
}

Train :: proc(adalineUnit: ^Adaline, trainData: ^MnistData, epochs: int)
{
    errorPerEpoch := make([]f32, epochs)
    
    for i := 0; i < epochs; i+=1
    {
        totalError: f32 = 0
        for j := 0; j < int(trainData.imageHeader.count); j+=1
        {
            randomIdx := rand.int_range(0, int(trainData.imageHeader.count))
    
            pixels := GetImage(trainData, randomIdx)
            target : f32 = 1.0 if int(trainData.labels[randomIdx]) == adalineUnit.label else -1.0
    
            o: f32 = DotProduct(adalineUnit.weights[1:], pixels) + adalineUnit.weights[0]
    
            diff := target - o
            error := adalineUnit.eta * diff
            totalError += diff * diff
    
            for k := 0; k < len(adalineUnit.weights) - 1; k+=1
            {
                adalineUnit.weights[k + 1] += error * pixels[k]
            }
            adalineUnit.weights[0] += adalineUnit.eta * (target - o) //bo wejscie = 1
        }

        errorPerEpoch[i] = totalError
    }

    filename := fmt.tprintf("Bledy_%d.txt", adalineUnit.label)
    SaveToCSV(filename, errorPerEpoch)

    fmt.print("Done\n")
}

Classify :: proc(units: ^[10]Adaline, pixels: []f32) -> int
{
    best_label := -1
    max_output : f32 = -1e30

    for i in 0..<10 
    {
        output := DotProduct(units[i].weights[1:], pixels) + units[i].weights[0]
        
        if output > max_output
        {
            max_output = output
            best_label = i
        }
    }

    return best_label
}

testModel :: proc(units: ^[10]Adaline, testData: ^MnistData) 
{
    hit := 0
    total := int(testData.imageHeader.count)

    for i in 0..<total 
    {
        pixels := GetImage(testData, i)
        
        predicted := Classify(units, pixels)
        actual := int(testData.labels[i])

        fmt.printfln("Predicted: %d --- Actual: %d", predicted, actual)

        if predicted == actual
        {
            hit += 1
        } 
    }

    accuracy := f32(hit) / f32(total) * 100.0
    fmt.printfln("Accuracy: %.2f%%", 100.0 - accuracy)
}

SaveToCSV :: proc(filename: string, errors: []f32) {

    fd, _ := os.open(filename, os.O_WRONLY | os.O_CREATE | os.O_TRUNC)
    defer os.close(fd)

    w := os.to_writer(fd)

    writer: csv.Writer

    csv.writer_init(&writer, w)

    csv.write(&writer, []string{"Epoch", "Error Value"})

    for e, i in errors 
    {

        epoch_str := fmt.tprintf("%d", i)
        error_str := fmt.tprintf("%.2f", e)
        
        record := []string{epoch_str, error_str}
        csv.write(&writer, record)
    }
    csv.writer_flush(&writer)
}

main :: proc()
{
    trainData: MnistData = {}
    testData: MnistData = {}

    err := ReadMnistFile(&trainData, "Data/Train/train-images.idx3-ubyte", "Data/Train/train-labels.idx1-ubyte")
    err2 := ReadMnistFile(&testData, "Data/Test/t10k-images.idx3-ubyte", "Data/Test/t10k-labels.idx1-ubyte")

    if err != true
    {
        fmt.eprintfln("Error")
        return
    }

    if err2 != true
    {
        fmt.eprintfln("Error, test file")
        return
    }

    adalineUnits: [10]Adaline = {}

    for i := 0; i < 10; i+=1
    {
        InitAdalineUnit(&adalineUnits[i], i)
    }

    for i := 0; i < 10; i+=1
    {
        Train(&adalineUnits[i], &trainData, 15)
    }


    testModel(&adalineUnits, &testData)


    delete(trainData.images)
    delete(trainData.labels)
    delete(testData.images)
    delete(testData.labels)
}
