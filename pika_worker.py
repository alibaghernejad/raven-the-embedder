

import logging
import pika
import json
from Settings import Settings
from embedding_pipeline import process_embedding_task

settings = Settings()

def callback(ch, method, properties, body):
    event_json = body.decode()
    try:
        event = json.loads(event_json)
        task_id = event['Fields'][0]["TaskId"]
        process_embedding_task(task_id)
        ch.basic_ack(delivery_tag=method.delivery_tag)
    except Exception as e:
        try:
            logging.error(f"Error processing message: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as inner_e:
            logging.error(f"Error acknowledging message: {inner_e}")

def start_consumer():
    params = pika.URLParameters(settings.rabbitmq_broker_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=settings.embedding_task_queue, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=settings.embedding_task_queue, on_message_callback=callback)
    print(f"[*] Waiting for messages in queue '{settings.embedding_task_queue}'. To exit press CTRL+C")
    while True:
        try:
            channel.start_consuming()
        except EOFError:
            print("[!] EOFError: Lost connection to broker, retrying...")
            try:
                channel.stop_consuming()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
            import time
            time.sleep(2)
            # Reconnect
            params = pika.URLParameters(settings.rabbitmq_broker_url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=settings.embedding_task_queue, durable=True)
            channel.basic_qos(prefetch_count=1)
            channel.basic_consume(queue=settings.embedding_task_queue, on_message_callback=callback)
            print(f"[*] Waiting for messages in queue '{settings.embedding_task_queue}'. To exit press CTRL+C")
            continue
        except KeyboardInterrupt:
            print("\n[!] Consumer stopped by user.")
            try:
                channel.stop_consuming()
            except Exception:
                pass
            break
        except Exception as e:
            logging.error(f"Unexpected error in consumer: {e}")
            try:
                channel.stop_consuming()
            except Exception:
                pass
            try:
                connection.close()
            except Exception:
                pass
            break
    try:
        connection.close()
    except Exception:
        pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    start_consumer()
