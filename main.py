import os
import PIL
import json
import time
import torch
import shutil
import tempfile
import datetime
import numpy as np
from monai.metrics import ROCAUCMetric
from monai.utils import set_determinism
from monai.networks.nets import DenseNet121
from monai.apps import download_and_extract
from monai.data import decollate_batch, DataLoader
from monai.transforms import ( Activations,
                              EnsureChannelFirst,
                              AsDiscrete,
                              Compose,
                              LoadImage,
                              RandFlip,
                              RandRotate,
                              RandZoom,
                              ScaleIntensity,
)


def write_json(json_path, config):
    
    with open(json_path, 'w') as handle:
        json.dump(config, handle, indent = 4)
        
    return None

def main():
    # set up directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    root_dir = tempfile.mkdtemp() if directory is None else directory
    
    # download dataset
    resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
    md5 = "0bc7306e7427e00ad1c5526a6677552d"
    
    compressed_file = os.path.join(root_dir, "MedNIST.tar.gz")
    data_dir = os.path.join(root_dir, "MedNIST")
    if not os.path.exists(data_dir):
        download_and_extract(resource, compressed_file, root_dir, md5)
        
    
    # set seed point
    set_determinism(seed=1234)
    
    # reformulate data
    class_names = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))
    num_class = len(class_names)
    image_files = [
        [os.path.join(data_dir, class_names[i], x) for x in os.listdir(os.path.join(data_dir, class_names[i]))]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    
    print(f"Total image count: {num_total}")
    print(f"Image dimensions: {image_width} x {image_height}")
    print(f"Label names: {class_names}")
    print(f"Label counts: {num_each}")
    
    
    # prepare training dataset
    
    val_frac = 0.1
    test_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)
    
    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]
    
    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]
    test_x = [image_files_list[i] for i in test_indices]
    test_y = [image_class[i] for i in test_indices]
    
    print(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")
    
    # set the transformations
    
    train_transforms = Compose(
        [
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )
    
    val_transforms = Compose([LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])
    
    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])
    
    class MedNISTDataset(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms
    
        def __len__(self):
            return len(self.image_files)
    
        def __getitem__(self, index):
            return self.transforms(self.image_files[index]), self.labels[index]
    
    
    train_ds = MedNISTDataset(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)
    
    val_ds = MedNISTDataset(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)
    
    test_ds = MedNISTDataset(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)
    
    # set the network topology and optimzation
    device = torch.device("cuda")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    max_epochs = 2
    val_interval = 1
    auc_metric = ROCAUCMetric()
    
    
    # model trainin
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []
    
    # logs setting
    current_device = torch.cuda.current_device() 
    n_device = torch.cuda.device_count() 
    name_device = torch.cuda.get_device_name(0)
    exec_time = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    exec_filename = 'TestLog_'+exec_time+'.json'
    
    st_time = time.time()
    dic_log = {}
    # training loop
    if len(train_x)>0:
        if n_device>0 and len(name_device)>0:
            for epoch in range(max_epochs):
                print("-" * 10)
                print(f"working on {epoch + 1}/{max_epochs}")
                model.train()
                epoch_loss = 0
                step = 0
                for batch_data in train_loader:
                    step += 1
                    inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    epoch_len = len(train_ds) // train_loader.batch_size
                epoch_loss /= step
                epoch_loss_values.append(epoch_loss)
                print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
            
                if (epoch + 1) % val_interval == 0:
                    model.eval()
                    with torch.no_grad():
                        y_pred = torch.tensor([], dtype=torch.float32, device=device)
                        y = torch.tensor([], dtype=torch.long, device=device)
                        for val_data in val_loader:
                            val_images, val_labels = (
                                val_data[0].to(device),
                                val_data[1].to(device),
                            )
                            y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                            y = torch.cat([y, val_labels], dim=0)
                        y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                        y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                        auc_metric(y_pred_act, y_onehot)
                        result = auc_metric.aggregate()
                        auc_metric.reset()
                        del y_pred_act, y_onehot
                        metric_values.append(result)
                        acc_value = torch.eq(y_pred.argmax(dim=1), y)
                        acc_metric = acc_value.sum().item() / len(acc_value)
                        if result > best_metric:
                            best_metric = result
                            best_metric_epoch = epoch + 1
                            torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                            print("saved new best metric model")
                            
                            
            end_time = time.time()
            elapsed_time = end_time-st_time
            dic_log['GPU_status'] = True
            dic_log['n_train_x'] = len(train_x)
            dic_log['n_train_y'] = len(train_y)
            dic_log['n_val_x'] = len(val_x)
            dic_log['n_val_y'] = len(val_y)
            dic_log['n_test_x'] = len(test_x)
            dic_log['n_test_y'] = len(test_y)
            dic_log['training time(seconds)'] = elapsed_time
            dic_log['temp_dir_saved_data'] = root_dir
        else:
            print('\n'*10)
            print('GPU was not recognized!')
            dic_log['GPU_status'] = False
    else:
        dic_log['Error Message:'] = 'Data was not downloaded or not saved properly!'
        
    
    json_path = os.path.join(os.getcwd(), exec_filename)
    write_json(json_path, dic_log)
    
    if directory is None:
        shutil.rmtree(root_dir)
    print('\n'*10)
    print('logs were saved successfully!')
    return None


if __name__ == '__main__':
    main()