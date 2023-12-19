use ort::{GraphOptimizationLevel, Session};
use ndarray::{s, Array3, Array};
use std::{fs::File, collections::HashMap};
use serde_json::from_reader;

#[derive(serde::Deserialize)]
struct Word2Index {
    word_2_index: HashMap<String, i64>,
}

// load word2index dict
fn load_word2index(filename: &str) -> Word2Index {
    let file = File::open(filename).unwrap();
    let data: Word2Index = from_reader(file).unwrap();
    data
}

// load model
fn load_model(model_path: &str) -> ort::Result<Session> {
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_model_from_file(model_path)
}

// secentces one-hot encoding
fn preprocess_input(input: &str, word2index: &HashMap<String, i64>) -> Vec<i64> {
    let mut input_vector = Vec::new();
    for word in input.chars() {
        input_vector.push(word2index[&word.to_string()]);
    }

    while input_vector.len() < 38 {
        input_vector.push(0);
    }

    input_vector
}

// infer
fn infer(model: &Session, input_vector: Vec<i64>) ->  Result<String, ort::Error> {
    // input tensor, shape: (1, 1, 38) come from model input shape when you train model
    let input_tensor: Array3<i64> = Array::from_shape_vec((1, 1, 38), input_vector).unwrap();

    let outputs = model.run(ort::inputs!{"input" => input_tensor.view()}?)?;

    let output = outputs["output"]
            .extract_tensor::<f32>()?
            .view()
            .t()
            .slice(s![.., 0])
            .into_owned();

    let argmx = output.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;

    // Labels
    let label = match argmx {
        0 => "finance",
        1 => "realty",
        2 => "stocks",
        3 => "education",
        4 => "science",
        5 => "society",
        6 => "politics",
        7 => "sports",
        8 => "game",
        9 => "entertainment",
        _ => "unknown",
    };
    
    Result::Ok(label.to_string())
}

fn main() -> ort::Result<()> {
    let start_time = std::time::Instant::now();
    let word2index = load_word2index("dict.json").word_2_index;

    let input = ["我们一起去打篮球吧！", "沈腾和马丽的新电影《独行月球》很好看", "昨天玩游戏，完了一整天",
    "现在的高考都已经开始分科考试了。", "中方：佩洛西如赴台将致严重后果", "现在的股票基金趋势很不好"];
    for i in input.iter() {
        let input_vector = preprocess_input(i, &word2index);
        let label = infer(&load_model("model.onnx")?, input_vector)?;
        print!("text:{},Label: {}\n", i,label);
    }
    let end_time = std::time::Instant::now();
    println!("time: {:?}", end_time.duration_since(start_time));
    Ok(())
}