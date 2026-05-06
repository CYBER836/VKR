import pandas as pd
import numpy as np
import dpkt
from collections import defaultdict
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_features_dpkt(pcap_path: str, output_csv: str):

    logger.info(f"Обработка: {pcap_path}")
    logger.info("Используется dpkt...")
    
    start_time = time.time()
    
    # Открываем PCAP
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        
        flows = defaultdict(lambda: {
            'timestamps': [],
            'sizes': [],
            'fwd_sizes': [],
            'bwd_sizes': [],
            'flags': [],
            'dst_port': None,
            'src_ip': None,
            'dst_ip': None
        })
        
        count = 0
        
        for timestamp, buf in pcap:
            count += 1
            
            if count % 50000 == 0:
                elapsed = time.time() - start_time
                logger.info(f"Обработано {count} пакетов... ({elapsed:.1f} сек)")
            
            try:
                # Ethernet -> IP
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                
                ip = eth.data
                src_ip = '.'.join(map(str, ip.src))
                dst_ip = '.'.join(map(str, ip.dst))
                protocol = ip.p
                size = len(buf)
                
                # TCP/UDP
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data
                    src_port = tcp.sport
                    dst_port = tcp.dport
                    flags = tcp.flags
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    src_port = udp.sport
                    dst_port = udp.dport
                    flags = 0
                else:
                    continue
                
                # Flow ID
                flow_id = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]) + [protocol])
                
                # Направление
                is_fwd = (src_ip, src_port) < (dst_ip, dst_port)
                
                # Сохраняем
                flow = flows[flow_id]
                flow['timestamps'].append(timestamp)
                flow['sizes'].append(size)
                flow['flags'].append(flags)
                
                if flow['dst_port'] is None:
                    flow['dst_port'] = dst_port if is_fwd else src_port
                    flow['src_ip'] = src_ip if is_fwd else dst_ip
                    flow['dst_ip'] = dst_ip if is_fwd else src_ip
                
                if is_fwd:
                    flow['fwd_sizes'].append(size)
                else:
                    flow['bwd_sizes'].append(size)
                    
            except Exception:
                continue
    
    elapsed = time.time() - start_time
    logger.info(f"Чтение завершено: {count} пакетов за {elapsed:.1f} сек")
    logger.info(f"Уникальных flows: {len(flows)}")
    
    # Расчёт признаков
    logger.info("Расчёт 16 признаков...")
    
    features_list = []
    
    for flow_id, flow in flows.items():
        if len(flow['packets']) < 2:
            continue
        
        timestamps = flow['timestamps']
        flow_duration = max(timestamps) - min(timestamps)
        if flow_duration == 0:
            flow_duration = 1e-6
        
        sizes = flow['sizes']
        fwd_sizes = flow['fwd_sizes']
        bwd_sizes = flow['bwd_sizes']
        all_flags = flow['flags']
        
        feat = {
            'Destination Port': flow['dst_port'] or 0,
            'Flow Duration': flow_duration,
            'Total Fwd Packets': len(fwd_sizes),
            'Total Backward Packets': len(bwd_sizes),
            'Fwd Packet Length Mean': np.mean(fwd_sizes) if fwd_sizes else 0,
            'Bwd Packet Length Mean': np.mean(bwd_sizes) if bwd_sizes else 0,
            'Flow Bytes/s': sum(sizes) / flow_duration,
            'Flow Packets/s': len(sizes) / flow_duration,
            'Average Packet Size': np.mean(sizes) if sizes else 0,
            'FIN Flag Count': sum(1 for f in all_flags if f & 0x01),
            'SYN Flag Count': sum(1 for f in all_flags if f & 0x02),
            'RST Flag Count': sum(1 for f in all_flags if f & 0x04),
            'PSH Flag Count': sum(1 for f in all_flags if f & 0x08),
            'ACK Flag Count': sum(1 for f in all_flags if f & 0x10),
            'URG Flag Count': sum(1 for f in all_flags if f & 0x20),
            'Flow ID': f"{flow_id[0][0]}:{flow_id[0][1]}-{flow_id[1][0]}:{flow_id[1][1]}",
            'Source IP': flow_id[0][0],
            'Destination IP': flow_id[1][0]
        }
        
        features_list.append(feat)
    
    df = pd.DataFrame(features_list)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Сохраняем
    df.to_csv(output_csv, index=False)
    
    total_time = time.time() - start_time
    logger.info(f"✅ Готово за {total_time:.1f} сек. Сохранено {len(df)} flows")
    return df


# ====== ЗАПУСК ======
PCAP_PATH = r"C:\\Паша\\Проекты\\VKR\\attack_traffic.pcap"
OUTPUT_CSV = r"C:\\Паша\\Проекты\\VKR\\extracted_16_features.csv"

print("=" * 60)
print("ИЗВЛЕЧЕНИЕ (dpkt)")
print("=" * 60)
print(f"PCAP: {PCAP_PATH}")
print(f"Выход: {OUTPUT_CSV}")
print("=" * 60)

df = extract_features_dpkt(PCAP_PATH, OUTPUT_CSV)

print(f"\\Результат:")
print(f"   Flows: {len(df)}")
print(f"   Файл: {OUTPUT_CSV}")
